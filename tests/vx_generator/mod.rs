#[derive(Default)]
pub struct StructBuilder {
    pub name: String,
    pub fields: Vec<(String, String)>,
}

impl StructBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            fields: Vec::new(),
        }
    }

    pub fn add_field(&mut self, name: &str, ty: &str) -> &mut Self {
        self.fields.push((name.to_string(), ty.to_string()));
        self
    }

    pub fn build(&self) -> String {
        let mut s = format!("struct {} {{\n", self.name);
        for (f, t) in &self.fields {
            s.push_str(&format!("  {}: {},\n", f, t));
        }
        s.push_str("}\n");
        s
    }
}

#[derive(Default)]
pub struct FunctionBuilder {
    pub name: String,
    pub args: Vec<(String, String)>,
    pub ret_type: String,
    pub body: String,
}

impl FunctionBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            args: Vec::new(),
            ret_type: "()".to_string(),
            body: String::new(),
        }
    }

    pub fn add_arg(&mut self, name: &str, ty: &str) -> &mut Self {
        self.args.push((name.to_string(), ty.to_string()));
        self
    }

    pub fn set_return_type(&mut self, ret_type: &str) -> &mut Self {
        self.ret_type = ret_type.to_string();
        self
    }

    pub fn add_statement(&mut self, stmt: &str) -> &mut Self {
        self.body.push_str("    ");
        self.body.push_str(stmt);
        self.body.push('\n');
        self
    }

    pub fn build(&self) -> String {
        let args_str = self
            .args
            .iter()
            .map(|(n, t)| format!("{}: {}", n, t))
            .collect::<Vec<_>>()
            .join(", ");

        let mut s = format!("fn {}({})", self.name, args_str);
        if self.ret_type != "()" {
            s.push_str(&format!(" -> {}", self.ret_type));
        }
        s.push_str(" {\n");
        s.push_str(&self.body);
        s.push_str("}\n");
        s
    }
}

#[derive(Default)]
pub struct ModuleBuilder {
    pub structs: Vec<StructBuilder>,
    pub functions: Vec<FunctionBuilder>,
}

impl ModuleBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_struct(&mut self, s: StructBuilder) -> &mut Self {
        self.structs.push(s);
        self
    }

    pub fn add_function(&mut self, f: FunctionBuilder) -> &mut Self {
        self.functions.push(f);
        self
    }

    pub fn build(&self) -> String {
        let mut s = String::new();
        for st in &self.structs {
            s.push_str(&st.build());
            s.push('\n');
        }
        for f in &self.functions {
            s.push_str(&f.build());
            s.push('\n');
        }
        s
    }
}
